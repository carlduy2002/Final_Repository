import { Component, HostListener, OnInit } from '@angular/core';
import { UserStoreService } from './Services/user-store.service';
import { ApiService } from './Services/api.service';
import { NavigationEnd, Router } from '@angular/router';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit{

  title = 'HD-Shoe Store';
  role : string = "";
  marginLeft: number = 0;
  hideButton: string = 'none';
  showButton: string = 'block'
  isShow: boolean = false;
  private currentPage: string = '';


  constructor(
    private userStore:UserStoreService,
    private api:ApiService,
    private router: Router,
  ){}

  ngOnInit(): void {
    this.detectDevTools();

    this.userStore.getRoleFromStore().subscribe(res => {
      let roleFromToken = this.api.getRoleFromToken();
      this.role = res || roleFromToken;
    });
  }

  @HostListener('document:keydown', ['$event'])
  onKeyDown(event: KeyboardEvent) {
    if (event.ctrlKey && event.shiftKey && (event.key === 'I' || event.key === 'i')) {
      event.preventDefault(); // Prevent opening Chrome DevTools
    }
  }

  @HostListener('document:contextmenu', ['$event'])
  onRightClick(event: MouseEvent) {
    event.preventDefault(); // Prevent right-click menu
  }

  detectDevTools() {
    setInterval(() => {
      if (window.outerWidth - window.innerWidth > 100 || window.outerHeight - window.innerHeight > 100) {
        // DevTools are open
        this.handleDevToolsOpen();
      }
    }, 1000);
  }

  handleDevToolsOpen() {
    // Perform actions when DevTools are detected
    console.log("Developer Tools are open!");
    window.location.reload(); // Redirect to a different page
  }

  isLoginPage(): boolean {
    return this.router.url.includes('login') || this.router.url.includes('register');
    //  || this.router.url.includes('check-old-password') || this.router.url.includes('change-password');
  }
}
